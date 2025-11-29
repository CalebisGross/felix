"""
LLM-powered agent for the Felix Framework.

This module extends the base Agent class with language model capabilities,
enabling agents to process tasks using local LLM inference via LM Studio.

Key Features:
- Integration with LM Studio for local LLM inference
- Position-aware prompt engineering based on helix location
- Adaptive behavior based on geometric constraints
- Communication via spoke system with central coordination
- Built-in task processing and result sharing

The agent's behavior adapts based on its position on the helix:
- Top (wide): Broad exploration, high creativity
- Middle: Focused analysis, balanced processing
- Bottom (narrow): Precise synthesis, low temperature
"""

import time
import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.agents.agent import Agent, AgentState
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient, LLMResponse, RequestPriority, TokenAwareStreamController
from src.llm.multi_server_client import LMStudioClientPool
from src.llm.token_budget import TokenBudgetManager, TokenAllocation
from src.communication.central_post import Message, MessageType
from src.pipeline.chunking import ChunkedResult, ProgressiveProcessor, ContentSummarizer
from src.agents.prompt_optimization import PromptOptimizer, PromptMetrics, PromptContext

logger = logging.getLogger(__name__)


@dataclass
class LLMTask:
    """Task for LLM agent processing."""
    task_id: str
    description: str
    context: str = ""
    metadata: Optional[Dict[str, Any]] = None
    context_history: Optional[List[Dict[str, Any]]] = None  # Previous agent outputs
    knowledge_entries: Optional[List[Any]] = None  # Relevant knowledge from memory
    tool_instructions: str = ""  # Conditional tool instructions based on task requirements

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMResult:
    """Result from LLM agent processing with chunking support."""
    agent_id: str
    task_id: str
    content: str
    position_info: Dict[str, float]
    llm_response: LLMResponse
    processing_time: float
    timestamp: float
    confidence: float = 0.0  # Confidence score (0.0 to 1.0)
    processing_stage: int = 1  # Stage number in helix descent

    # Prompt and settings used for this processing
    system_prompt: str = ""  # System prompt sent to LLM
    user_prompt: str = ""  # User prompt sent to LLM
    temperature_used: float = 0.0  # Temperature setting used
    token_budget_allocated: int = 0  # Token budget allocated for this stage
    collaborative_context_count: int = 0  # Number of previous agent outputs used

    # Chunking support fields
    is_chunked: bool = False  # Whether this result contains chunked output
    chunk_results: Optional[List[ChunkedResult]] = None  # List of chunk results if chunked
    total_chunks: int = 1  # Total number of chunks (1 for non-chunked)
    chunking_strategy: Optional[str] = None  # Strategy used for chunking (progressive/streaming)
    summary_fallback: Optional[str] = None  # Summary if content was truncated
    full_content_available: bool = True  # Whether full content is available or summarized
    
    def __post_init__(self):
        if self.chunk_results is None and self.is_chunked:
            self.chunk_results = []
        elif self.chunk_results is None:
            self.chunk_results = []
    
    def get_full_content(self) -> str:
        """Get full content, combining chunks if necessary."""
        if not self.is_chunked:
            return self.content
        
        if not self.chunk_results:
            return self.content
        
        # Combine chunk content in order
        sorted_chunks = sorted(self.chunk_results, key=lambda x: x.chunk_index)
        combined_content = "".join(chunk.content_chunk for chunk in sorted_chunks)
        
        return combined_content if combined_content else self.content
    
    def get_content_summary(self) -> str:
        """Get content summary, preferring summary_fallback if available."""
        if self.summary_fallback and not self.full_content_available:
            return self.summary_fallback
        
        content = self.get_full_content()
        if len(content) <= 200:
            return content
        
        return content[:200] + "..."
    
    def add_chunk(self, chunk: ChunkedResult) -> None:
        """Add a chunk result to this LLM result."""
        if not self.is_chunked:
            self.is_chunked = True
            
        if self.chunk_results is None:
            self.chunk_results = []
            
        self.chunk_results.append(chunk)
        self.total_chunks = len(self.chunk_results)
    
    def is_complete(self) -> bool:
        """Check if all chunks are available for a chunked result."""
        if not self.is_chunked:
            return True
        
        if not self.chunk_results:
            return False
        
        # Check if we have a final chunk
        return any(chunk.is_final for chunk in self.chunk_results)
    
    def get_chunking_metadata(self) -> Dict[str, Any]:
        """Get metadata about the chunking process."""
        if not self.is_chunked:
            return {"chunked": False}
        
        return {
            "chunked": True,
            "total_chunks": self.total_chunks,
            "chunks_available": len(self.chunk_results) if self.chunk_results else 0,
            "strategy": self.chunking_strategy,
            "complete": self.is_complete(),
            "full_content_available": self.full_content_available,
            "has_summary_fallback": self.summary_fallback is not None
        }


class LLMAgent(Agent):
    """
    LLM-powered agent that processes tasks using language models.
    
    Extends the base Agent class with LLM capabilities, providing
    position-aware prompt engineering and adaptive behavior based
    on the agent's location on the helix.
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client, agent_type: str = "general",
                 temperature_range: Optional[tuple] = None, max_tokens: Optional[int] = None,
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 prompt_optimizer: Optional[PromptOptimizer] = None,
                 prompt_manager: Optional['PromptManager'] = None):
        """
        Initialize LLM agent.
        
        Args:
            agent_id: Unique identifier for the agent
            spawn_time: Time when agent becomes active (0.0 to 1.0)
            helix: Helix geometry for path calculation
            llm_client: LM Studio client or client pool for LLM inference
            agent_type: Agent specialization (research, analysis, synthesis, critic)
            temperature_range: (min, max) temperature based on helix position
            token_budget_manager: Optional budget manager for adaptive token allocation
            prompt_optimizer: Optional prompt optimization system for learning
            prompt_manager: Optional prompt manager for custom prompt templates
        """
        super().__init__(agent_id, spawn_time, helix)

        self.llm_client = llm_client
        self.agent_type = agent_type
        self.prompt_manager = prompt_manager
        
        # FIXED: Set appropriate defaults based on agent type (aligns with TokenBudgetManager)
        if temperature_range is None:
            if agent_type == "research":
                self.temperature_range = (0.4, 0.9)  # High creativity for exploration
            elif agent_type == "analysis":
                self.temperature_range = (0.2, 0.7)  # Balanced for processing
            elif agent_type == "synthesis":
                self.temperature_range = (0.1, 0.5)  # Lower for focused synthesis
            elif agent_type == "critic":
                self.temperature_range = (0.1, 0.6)  # Low-medium for critique
            else:
                self.temperature_range = (0.1, 0.9)  # Default fallback
        else:
            self.temperature_range = temperature_range
        
        if max_tokens is None:
            if agent_type == "research":
                self.max_tokens = 1000  # Expanded for comprehensive research findings
            elif agent_type == "analysis":
                self.max_tokens = 1000  # Expanded for detailed structured analysis
            elif agent_type == "synthesis":
                self.max_tokens = 20000  # Very large for comprehensive final output
            elif agent_type == "critic":
                self.max_tokens = 1000  # Expanded for thorough critique
            else:
                self.max_tokens = 1000  # Default fallback
        else:
            self.max_tokens = max_tokens
            
        self.token_budget_manager = token_budget_manager
        self.prompt_optimizer = prompt_optimizer
        
        # Initialize token budget if manager provided
        if self.token_budget_manager:
            self.token_budget_manager.initialize_agent_budget(agent_id, agent_type, self.max_tokens)
        
        # LLM-specific state
        self.processing_results: List[LLMResult] = []
        self.total_tokens_used = 0
        self.total_processing_time = 0.0
        self.processing_stage = 0  # Current processing stage in helix descent
        self._last_checkpoint_processed = -1  # Track last checkpoint for multi-stage processing

        # Token efficiency tracking for adaptive behavior
        self.token_efficiency_history: List[float] = []  # Ratio of used/allocated tokens per call
        self._token_adjustment_factor = 1.0  # Multiplicative adjustment based on efficiency
        self._token_overshoot_count = 0  # Count of times agent exceeded budget
        self._token_undershoot_count = 0  # Count of times agent used <70% of budget

        # Communication state
        self.shared_context: Dict[str, Any] = {}
        self.received_messages: List[Dict[str, Any]] = []

        # Emergent behavior tracking
        self.influenced_by: List[str] = []  # Agent IDs that influenced this agent
        self.influence_strength: Dict[str, float] = {}  # How much each agent influenced this one
        self.collaboration_history: List[Dict[str, Any]] = []  # History of collaborations

        # Feedback integration for self-improvement
        self.contribution_history: List[Dict[str, Any]] = []  # History of contribution evaluations
        self.synthesis_integration_rate = 0.0  # Running average of how often contributions are used
        self._total_feedback_received = 0  # Count of feedback messages received

    def get_adaptive_temperature(self, current_time: float) -> float:
        """
        Calculate temperature based on helix position.
        
        Higher temperature (more creative) at top of helix,
        lower temperature (more focused) at bottom.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Temperature value for LLM
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Invert depth ratio: top (0.0) = high temp, bottom (1.0) = low temp
        inverted_ratio = 1.0 - depth_ratio
        
        min_temp, max_temp = self.temperature_range
        temperature = min_temp + (max_temp - min_temp) * inverted_ratio
        
        return max(min_temp, min(max_temp, temperature))
    
    def calculate_confidence(self, current_time: float, content: str, stage: int, task: Optional[LLMTask] = None) -> float:
        """
        Calculate confidence score based on agent type, helix position, content quality, and collaboration.

        Agent types have different confidence ranges to ensure proper workflow:
        - Research agents: 0.3-0.6 (gather info, don't make final decisions)
        - Analysis agents: 0.4-0.8 (process info, prepare for synthesis)
        - Synthesis agents: 0.6-0.95 (create final output)
        - Critic agents: 0.5-0.8 (provide feedback)

        Collaborative bonus can add up to 0.15 additional confidence when agents
        synthesize multiple perspectives and build consensus.

        Args:
            current_time: Current simulation time
            content: Generated content to evaluate
            stage: Processing stage number

        Returns:
            Confidence score (0.0 to 1.0)
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Base confidence range based on agent type
        if self.agent_type == "research":
            # Research agents max out at 0.6 - they gather info, don't make final decisions
            base_confidence = 0.3 + (depth_ratio * 0.3)  # 0.3-0.6 range
            max_confidence = 0.6
        elif self.agent_type == "analysis":
            # Analysis agents: 0.4-0.8 - process info but don't synthesize
            # Raised from 0.7 to allow collaborative analysis to exceed consensus threshold
            base_confidence = 0.4 + (depth_ratio * 0.4)  # 0.4-0.8 range
            max_confidence = 0.8
        elif self.agent_type == "synthesis":
            # Synthesis agents: 0.6-0.95 - create final comprehensive output
            base_confidence = 0.6 + (depth_ratio * 0.35)  # 0.6-0.95 range
            max_confidence = 0.95
        elif self.agent_type == "critic":
            # Critic agents: 0.5-0.8 - provide feedback and validation
            base_confidence = 0.5 + (depth_ratio * 0.3)  # 0.5-0.8 range
            max_confidence = 0.8
        else:
            # Default fallback
            base_confidence = 0.3 + (depth_ratio * 0.4)
            max_confidence = 0.7
        
        # Content quality bonus (up to 0.1 additional)
        content_quality = self._analyze_content_quality(content)
        content_bonus = content_quality * 0.1
        
        # Processing stage bonus (up to 0.05 additional)
        stage_bonus = min(stage * 0.005, 0.05)

        # Historical consistency bonus (up to 0.05 additional)
        consistency_bonus = self._calculate_consistency_bonus() * 0.05

        # Collaborative context bonus (up to 0.15 additional) - NEW
        # Rewards agents that synthesize multiple perspectives and build consensus
        collaborative_bonus = 0.0
        if task is not None:
            collaborative_bonus = self._calculate_collaborative_bonus(task, current_time) * 0.15

        total_confidence = base_confidence + content_bonus + stage_bonus + consistency_bonus + collaborative_bonus
        
        # Store debug info for potential display
        self._last_confidence_breakdown = {
            "base_confidence": base_confidence,
            "content_bonus": content_bonus,
            "stage_bonus": stage_bonus,
            "consistency_bonus": consistency_bonus,
            "collaborative_bonus": collaborative_bonus,
            "total_before_cap": total_confidence,
            "max_confidence": max_confidence,
            "final_confidence": min(max(total_confidence, 0.0), max_confidence)
        }
        
        return min(max(total_confidence, 0.0), max_confidence)
    
    def _analyze_content_quality(self, content: str) -> float:
        """
        Analyze content quality using multiple heuristics.
        
        Args:
            content: Content to analyze
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not content or len(content.strip()) == 0:
            return 0.0
        
        content_lower = content.lower()
        quality_score = 0.0
        
        # Length appropriateness (0.25 weight)
        content_length = len(content)
        if 100 <= content_length <= 2000:
            length_score = 1.0
        elif content_length < 100:
            length_score = content_length / 100.0
        else:  # Very long content
            length_score = max(0.3, 2000.0 / content_length)
        quality_score += length_score * 0.25
        
        # Structure indicators (0.25 weight)
        structure_indicators = [
            '\n\n' in content,  # Paragraphs
            '.' in content,     # Sentences
            any(word in content_lower for word in ['analysis', 'research', 'conclusion', 'summary']),
            content.count('.') >= 3,  # Multiple sentences
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)
        quality_score += structure_score * 0.25
        
        # Content depth indicators (0.25 weight)
        depth_indicators = [
            any(word in content_lower for word in ['because', 'therefore', 'however', 'moreover', 'furthermore']),
            any(word in content_lower for word in ['data', 'evidence', 'study', 'research', 'analysis']),
            any(word in content_lower for word in ['consider', 'suggest', 'indicate', 'demonstrate']),
            len(content.split()) > 50,  # Substantial word count
        ]
        depth_score = sum(depth_indicators) / len(depth_indicators)
        quality_score += depth_score * 0.25
        
        # Specificity indicators (0.25 weight)
        specificity_indicators = [
            any(char.isdigit() for char in content),  # Contains numbers/data
            content.count(',') > 2,  # Complex sentences with details
            any(word in content_lower for word in ['specific', 'particular', 'detail', 'example']),
            '"' in content or "'" in content,  # Quotes or citations
        ]
        specificity_score = sum(specificity_indicators) / len(specificity_indicators)
        quality_score += specificity_score * 0.25
        
        return min(quality_score, 1.0)
    
    def _calculate_consistency_bonus(self) -> float:
        """
        Calculate consistency bonus based on confidence history stability.
        
        Returns:
            Consistency bonus (0.0 to 1.0)
        """
        if len(self._confidence_history) < 3:
            return 0.5  # Neutral for insufficient data
        
        # Calculate confidence variance (lower variance = more consistent)
        recent_confidences = self._confidence_history[-3:]
        avg_confidence = sum(recent_confidences) / len(recent_confidences)
        variance = sum((c - avg_confidence) ** 2 for c in recent_confidences) / len(recent_confidences)
        
        # Convert variance to consistency bonus (lower variance = higher bonus)
        consistency_bonus = max(0.0, 1.0 - (variance * 10))  # Scale variance appropriately
        
        return min(consistency_bonus, 1.0)

    def _calculate_collaborative_bonus(self, task: LLMTask, current_time: float) -> float:
        """
        Calculate confidence bonus for leveraging collaborative context.

        This rewards agents that synthesize multiple perspectives and
        build on previous work, enabling confidence to rise over time.

        Args:
            task: Current task being processed
            current_time: Current simulation time

        Returns:
            Bonus factor from 0.0 to 1.0
        """
        if not hasattr(task, 'context_history') or not task.context_history:
            return 0.0

        context_count = len(task.context_history)

        # Bonus for number of collaborators (0.0-0.4)
        # More perspectives = more confidence in synthesis
        count_bonus = min(context_count / 5.0, 0.4)

        # Bonus for consensus (0.0-0.3)
        # Low variance in previous confidence = team converging
        confidences = [c.get('confidence', 0.5) for c in task.context_history]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            consensus_bonus = max(0, 0.3 - variance * 3)  # Low variance = high bonus
        else:
            consensus_bonus = 0.0

        # Bonus for synthesis agents (0.0-0.3)
        # Synthesis agents should have highest confidence
        synthesis_bonus = 0.3 if self.agent_type == "synthesis" else 0.0

        # Total normalized to 0.0-1.0
        total = (count_bonus + consensus_bonus + synthesis_bonus) / 1.0
        return min(total, 1.0)

    def _learn_from_token_usage(self, allocated: int, used: int) -> None:
        """
        Learn from token usage patterns and adjust future behavior.

        Tracks efficiency ratios and adjusts the agent's token adjustment factor
        to proactively request appropriate budgets in future calls.

        Args:
            allocated: Tokens allocated for this generation
            used: Tokens actually used
        """
        if allocated == 0:
            return

        ratio = used / allocated
        self.token_efficiency_history.append(ratio)

        # Track overshoots and undershoots
        if ratio > 1.0:
            self._token_overshoot_count += 1
            logger.debug(f"Agent {self.agent_id} overshot budget: {used}/{allocated} "
                        f"({ratio:.1%}) - overshoot count: {self._token_overshoot_count}")
        elif ratio < 0.7:
            self._token_undershoot_count += 1
            logger.debug(f"Agent {self.agent_id} underused budget: {used}/{allocated} "
                        f"({ratio:.1%}) - undershoot count: {self._token_undershoot_count}")

        # Adaptive adjustment after sufficient history
        if len(self.token_efficiency_history) >= 3:
            # Calculate rolling average of last 3 calls
            recent_ratios = self.token_efficiency_history[-3:]
            avg_ratio = sum(recent_ratios) / len(recent_ratios)

            # Adjust factor based on consistent patterns
            if avg_ratio > 1.1:
                # Consistently over by >10% - request less next time
                self._token_adjustment_factor = max(0.7, self._token_adjustment_factor * 0.9)
                logger.info(f"Agent {self.agent_id} learning: consistently overshooting, "
                           f"adjustment factor now {self._token_adjustment_factor:.2f}")
            elif avg_ratio < 0.7:
                # Consistently under 70% - can use more
                self._token_adjustment_factor = min(1.2, self._token_adjustment_factor * 1.1)
                logger.info(f"Agent {self.agent_id} learning: consistently undershooting, "
                           f"adjustment factor now {self._token_adjustment_factor:.2f}")
            elif 0.8 <= avg_ratio <= 1.0:
                # Sweet spot - gradually move toward 1.0
                self._token_adjustment_factor = 0.9 * self._token_adjustment_factor + 0.1 * 1.0
                logger.debug(f"Agent {self.agent_id} in efficient range, "
                            f"adjustment factor: {self._token_adjustment_factor:.2f}")

    def get_token_efficiency(self) -> float:
        """
        Get agent's token efficiency metric.

        Returns average efficiency ratio from recent history, or 1.0 if no history.
        Efficiency < 1.0 means agent is concise, > 1.0 means agent overshoots.

        Returns:
            Average efficiency ratio (used/allocated)
        """
        if not self.token_efficiency_history:
            return 1.0  # No history yet, assume perfect efficiency

        # Return average of last 5 calls (or all if fewer)
        recent_history = self.token_efficiency_history[-5:]
        return sum(recent_history) / len(recent_history)

    def get_adjusted_token_budget(self, base_budget: int) -> int:
        """
        Get token budget adjusted by learned efficiency factor.

        Args:
            base_budget: Base token budget from TokenBudgetManager

        Returns:
            Adjusted budget based on agent's learned efficiency
        """
        adjusted = int(base_budget * self._token_adjustment_factor)

        # Ensure reasonable bounds (50%-150% of base)
        min_budget = int(base_budget * 0.5)
        max_budget = int(base_budget * 1.5)

        return max(min_budget, min(adjusted, max_budget))

    # Helical checkpoint system for continuous communication
    HELICAL_CHECKPOINTS = [0.0, 0.3, 0.5, 0.7, 0.9]

    def should_process_at_checkpoint(self, current_time: float) -> bool:
        """
        Check if agent has crossed a checkpoint threshold and should make an LLM call.

        This enables continuous communication as agents descend the helix:
        - Spawn (0.0): Initial exploration
        - Checkpoint 0.3: Early analysis
        - Checkpoint 0.5: Mid-analysis
        - Checkpoint 0.7: Synthesis preparation
        - Checkpoint 0.9: Final synthesis

        Uses existing self._progress attribute from base Agent class that
        tracks position along helix (0.0 = top/wide, 1.0 = bottom/narrow).

        Args:
            current_time: Current simulation time

        Returns:
            True if agent has crossed a new checkpoint since last LLM call
        """
        # Get current checkpoint index based on progress
        current_checkpoint_index = -1

        # Find highest checkpoint agent has passed
        for i, checkpoint in enumerate(self.HELICAL_CHECKPOINTS):
            if self._progress >= checkpoint:
                current_checkpoint_index = i

        # Should process if we've passed a new checkpoint
        should_process = current_checkpoint_index > self._last_checkpoint_processed

        return should_process

    def get_current_checkpoint(self) -> float:
        """
        Get the checkpoint value agent is currently at or has passed.

        Returns:
            Checkpoint value (0.0, 0.3, 0.5, 0.7, or 0.9)
        """
        for i in range(len(self.HELICAL_CHECKPOINTS) - 1, -1, -1):
            if self._progress >= self.HELICAL_CHECKPOINTS[i]:
                return self.HELICAL_CHECKPOINTS[i]
        return 0.0

    def mark_checkpoint_processed(self) -> None:
        """
        Mark current checkpoint as processed after making an LLM call.

        Updates _last_checkpoint_processed to prevent redundant calls at same checkpoint.
        """
        # Find current checkpoint index
        for i, checkpoint in enumerate(self.HELICAL_CHECKPOINTS):
            if self._progress >= checkpoint:
                self._last_checkpoint_processed = i

    def _build_strict_rules(self, task: LLMTask) -> List[str]:
        """
        Build strict rules based on available context to prevent agent comprehension failures.

        Args:
            task: Task object containing context

        Returns:
            List of strict rule strings
        """
        strict_rules = []

        # Rule 1: Tool usage
        if hasattr(task, 'tool_instructions') and task.tool_instructions:
            strict_rules.append(
                "🛠️ TOOL RULE: Tool instructions are provided below. USE them. "
                "DO NOT request different tools or claim you lack tools."
            )

        # Rule 2: Web search data
        if hasattr(task, 'knowledge_entries') and task.knowledge_entries:
            web_search_present = any(
                hasattr(k, 'domain') and k.domain == "web_search"
                for k in task.knowledge_entries
            )
            if web_search_present:
                strict_rules.append(
                    "🔍 WEB SEARCH RULE: Web search data is ALREADY PROVIDED below. "
                    "DO NOT write 'WEB_SEARCH_NEEDED:' - you already have the data. "
                    "Use the existing knowledge instead."
                )

        # Rule 3: Previous agent outputs
        if hasattr(task, 'context_history') and task.context_history:
            strict_rules.append(
                f"👥 COLLABORATION RULE: {len(task.context_history)} agent(s) have already contributed. "
                f"Build on their work. DO NOT repeat what they found or said."
            )

        return strict_rules

    def _build_response_format(self) -> str:
        """
        Build mandatory response format instructions.

        Returns:
            Formatted response format string
        """
        return """
🎯 MANDATORY RESPONSE FORMAT:

Before your main response, write this acknowledgment line:
CONTEXT_USED: [brief summary of what context/knowledge/tools you used]

Examples:
- CONTEXT_USED: Web search data from 2 sources, previous research outputs
- CONTEXT_USED: File operation tools
- CONTEXT_USED: None (first agent, no prior context)

Then provide your response.

This acknowledgment ensures you've reviewed available resources before responding.
"""

    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create system prompt that adapts to agent's helix position with token budget.
        Enhanced with prompt optimization that learns from performance metrics.
        
        Args:
            task: Task to process
            current_time: Current simulation time
            
        Returns:
            Tuple of (position-aware system prompt, token budget for this stage)
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Determine prompt context based on agent type and position
        prompt_context = self._get_prompt_context(depth_ratio)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Default fallback
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget
        
        # Add shared context from other agents
        context_summary = ""
        if self.shared_context:
            context_summary = "\n\nShared Context from Other Agents:\n"
            for key, value in self.shared_context.items():
                context_summary += f"- {key}: {value}\n"

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

        # NEW: Add existing concept definitions for terminology consistency
        concepts_summary = ""
        if hasattr(task, 'existing_concepts') and task.existing_concepts:
            concepts_summary = "\n\n=== Existing Concept Definitions ===\n"
            concepts_summary += "The following concepts have already been defined by other agents in this workflow.\n"
            concepts_summary += "Please use these definitions consistently instead of redefining them:\n\n"
            concepts_summary += task.existing_concepts
            concepts_summary += "\n\nIMPORTANT: Reference these existing concepts when relevant. "
            concepts_summary += "Only define new concepts if they are not already covered above.\n"

        # Generate prompt ID for optimization tracking
        prompt_id = f"{self.agent_type}_{prompt_context.value}_stage_{self.processing_stage}"
        
        # Check if we have an optimized prompt available
        optimized_prompt = None
        if self.prompt_optimizer:
            recommendations = self.prompt_optimizer.get_optimization_recommendations(prompt_context)
            if recommendations.get("best_prompts"):
                best_prompt = recommendations["best_prompts"][0]
                if best_prompt[1] > 0.7:  # High performance threshold
                    optimized_prompt = best_prompt[0]
                    logger.debug(f"Using optimized prompt for {prompt_id} (score: {best_prompt[1]:.3f})")
        
        # Create base system prompt
        if optimized_prompt:
            base_prompt = optimized_prompt
        else:
            # CRITICAL: Include conditional tool instructions if available
            tool_instructions = ""
            if hasattr(task, 'tool_instructions') and task.tool_instructions:
                tool_instructions = f"\n\n{task.tool_instructions}"
                logger.debug(f"Including tool instructions in prompt ({len(task.tool_instructions)} chars)")

            base_prompt = self.llm_client.create_agent_system_prompt(
                agent_type=self.agent_type,
                position_info=position_info,
                task_context=f"{task.context}{context_summary}{knowledge_summary}{concepts_summary}{tool_instructions}"
            )

        # NEW: Inject Context Awareness Protocol (Hybrid Imperative Prompting)
        # This forces agents to comprehend and use available context instead of ignoring it
        protocol_section = ""

        # Step 1: Add context inventory if available
        if hasattr(task, 'context_inventory') and task.context_inventory:
            protocol_section += "\n\n" + task.context_inventory + "\n"

        # Step 2: Add strict rules based on available resources
        strict_rules = self._build_strict_rules(task)
        if strict_rules:
            protocol_section += "\n🚨 STRICT RULES (FOLLOW THESE CAREFULLY):\n"
            for i, rule in enumerate(strict_rules, 1):
                protocol_section += f"  {i}. {rule}\n"

        # Step 3: Add mandatory response format
        protocol_section += self._build_response_format()

        # Inject protocol into base prompt
        if protocol_section:
            base_prompt = base_prompt + protocol_section
            logger.debug(f"Injected context awareness protocol ({len(protocol_section)} chars)")

        # Calculate temperature for metadata
        temperature = self.get_adaptive_temperature(current_time)

        # Build metadata section
        metadata_parts = ["\n\n=== Processing Parameters ==="]

        # Add temperature guidance
        if temperature < 0.3:
            temp_guidance = "PRECISE mode (low creativity): Focus on accuracy and specific details"
        elif temperature < 0.5:
            temp_guidance = "BALANCED mode: Mix precision with moderate exploration"
        elif temperature < 0.7:
            temp_guidance = "EXPLORATORY mode: Consider diverse perspectives and connections"
        else:
            temp_guidance = "CREATIVE mode (high creativity): Generate novel insights and broad exploration"

        metadata_parts.append(f"Temperature: {temperature:.2f} - {temp_guidance}")

        # Add token budget information
        if token_allocation:
            metadata_parts.append(f"Token Budget: {stage_token_budget} tokens (remaining: {token_allocation.remaining_budget})")
            metadata_parts.append(f"Compression Target: {token_allocation.compression_ratio:.0%}")
            metadata_parts.append(f"Output Guidance: {token_allocation.style_guidance}")
        else:
            metadata_parts.append(f"Token Budget: {stage_token_budget} tokens")

        # Add position information
        metadata_parts.append(f"Helix Position: {depth_ratio:.1%} depth (stage {self.processing_stage + 1})")

        # Combine everything
        metadata_section = "\n".join(metadata_parts)
        enhanced_prompt = base_prompt + metadata_section

        return enhanced_prompt, stage_token_budget
    
    def _get_prompt_context(self, depth_ratio: float) -> PromptContext:
        """
        Determine prompt context based on agent type and helix position.
        
        Args:
            depth_ratio: Agent's depth ratio on helix (0.0 = top, 1.0 = bottom)
            
        Returns:
            PromptContext enum value
        """
        if self.agent_type == "research":
            return PromptContext.RESEARCH_EARLY if depth_ratio < 0.3 else PromptContext.RESEARCH_MID
        elif self.agent_type == "analysis":
            return PromptContext.ANALYSIS_MID if depth_ratio < 0.7 else PromptContext.ANALYSIS_LATE
        elif self.agent_type == "synthesis":
            return PromptContext.SYNTHESIS_LATE
        elif self.agent_type == "critic":
            return PromptContext.GENERAL  # Critics can work at any stage
        else:
            return PromptContext.GENERAL
    
    def _record_prompt_metrics(self, prompt_text: str, prompt_context: PromptContext, 
                              result: LLMResult) -> None:
        """
        Record prompt performance metrics for optimization learning.
        
        Args:
            prompt_text: The full system prompt used
            prompt_context: Context category for the prompt
            result: LLM result containing performance data
        """
        if not self.prompt_optimizer:
            return
        
        # Calculate token efficiency
        tokens_used = getattr(result.llm_response, 'tokens_used', 0)
        token_efficiency = min(result.confidence, 0.8) if tokens_used > 0 else 0.0
        
        # Determine if truncation occurred (approximate)
        truncation_occurred = (
            tokens_used >= self.max_tokens * 0.95 or  # Used most of token budget
            result.llm_response.content.endswith("...") or  # Ends with ellipsis
            len(result.llm_response.content) < 50  # Very short response
        )
        
        # Create metrics
        metrics = PromptMetrics(
            output_quality=result.confidence,  # Use confidence as proxy for quality
            confidence=result.confidence,
            completion_time=result.processing_time,
            token_efficiency=token_efficiency,
            truncation_occurred=truncation_occurred,
            context=prompt_context
        )
        
        # Generate prompt ID for tracking
        prompt_id = f"{self.agent_type}_{prompt_context.value}_stage_{self.processing_stage}"
        
        # Record metrics
        optimization_result = self.prompt_optimizer.record_prompt_execution(
            prompt_id, prompt_text, metrics
        )
        
        if optimization_result.get("optimization_triggered"):
            logger.info(f"Prompt optimization triggered for {prompt_id}")

    def _build_collaborative_prompt(self, original_task: str,
                                    context_history: List[Dict[str, Any]],
                                    agent_type: str) -> str:
        """
        Build collaborative prompt incorporating previous agent outputs.

        This creates prompts that reference and build upon previous agents' work,
        enabling true multi-agent collaboration rather than isolated processing.

        Args:
            original_task: The original user task
            context_history: List of previous agent outputs
            agent_type: Type of current agent (research, critic, analysis, synthesis)

        Returns:
            Enhanced prompt that incorporates collaborative context
        """
        if not context_history:
            return original_task

        # Build context summary from previous agents
        context_parts = [f"Original Task: {original_task}", "\nPrevious Agent Outputs:"]

        for i, entry in enumerate(context_history, 1):
            prev_agent_type = entry.get('agent_type', 'unknown')
            prev_response = entry.get('response', '')
            prev_confidence = entry.get('confidence', 0.0)

            # Truncate very long responses but allow more detail with higher token budgets
            if len(prev_response) > 1000:
                prev_response = prev_response[:1000] + "..."

            context_parts.append(
                f"\n{i}. {prev_agent_type.upper()} Agent (confidence: {prev_confidence:.2f}):\n{prev_response}"
            )

        # Add concise collaborative instructions
        context_parts.append("\n---")
        context_parts.append("\n**Collaborative Context**: Build upon previous agents' work. DO NOT repeat - ADD NEW insights.")

        if agent_type == "research":
            context_parts.append(
                "\n\n**Research Focus**: Identify gaps in previous findings and explore them with NEW discoveries."
            )
        elif agent_type == "critic":
            # Add agent-specific focus for diversity
            import random
            focus_areas = ["logic", "accuracy", "completeness", "clarity", "novelty"]
            random.seed(hash(self.agent_id))
            focus = random.choice(focus_areas)

            context_parts.append(
                f"\n\n**Critic Focus ({focus})**: Evaluate collective outputs for consistency and quality. "
                "Higher confidence when agents agree."
            )
        elif agent_type == "analysis":
            context_parts.append(
                "\n\n**Analysis Focus**: Synthesize findings into patterns. Resolve contradictions using evidence."
            )
        elif agent_type == "synthesis":
            context_parts.append(
                "\n\n**Synthesis Focus**: Integrate ALL findings into comprehensive conclusion. Highest confidence from consensus."
            )

        return "\n".join(context_parts)

    async def process_task_with_llm_async(self, task: LLMTask, current_time: float, 
                                         priority: RequestPriority = RequestPriority.NORMAL) -> LLMResult:
        """
        Asynchronously process task using LLM with position-aware prompting.
        
        Args:
            task: Task to process
            current_time: Current simulation time
            priority: Request priority for LLM processing
            
        Returns:
            LLM processing result
        """
        start_time = time.perf_counter()

        # Get position-aware prompts, token budget, and temperature
        system_prompt, stage_token_budget = self.create_position_aware_prompt(task, current_time)
        temperature = self.get_adaptive_temperature(current_time)
        position_info = self.get_position_info(current_time)

        # Apply learned adjustment factor to budget
        adjusted_budget = self.get_adjusted_token_budget(stage_token_budget)

        # Ensure adjusted budget doesn't exceed agent's max_tokens
        effective_token_budget = min(adjusted_budget, self.max_tokens)

        logger.debug(f"Agent {self.agent_id} async token budget: base={stage_token_budget}, "
                    f"adjusted={adjusted_budget}, effective={effective_token_budget} "
                    f"(adjustment_factor={self._token_adjustment_factor:.2f})")

        # Build collaborative prompt if context history available
        if hasattr(task, 'context_history') and task.context_history:
            logger.info(f"Agent {self.agent_id} using collaborative context: "
                       f"{len(task.context_history)} previous outputs")
            user_prompt = self._build_collaborative_prompt(
                original_task=task.description,
                context_history=task.context_history,
                agent_type=self.agent_type
            )
        else:
            logger.info(f"Agent {self.agent_id} processing without collaborative context")
            user_prompt = task.description

        # Process with LLM using coordinated token budget (ASYNC)
        # Use multi-server client pool if available, otherwise use regular client
        if isinstance(self.llm_client, LMStudioClientPool):
            llm_response = await self.llm_client.complete_for_agent_type(
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=effective_token_budget,
                priority=priority
            )
        else:
            llm_response = await self.llm_client.complete_async(
                agent_id=self.agent_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=effective_token_budget,
                priority=priority
            )
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Increment processing stage
        self.processing_stage += 1

        # Calculate confidence based on position, content, and collaborative context
        confidence = self.calculate_confidence(current_time, llm_response.content, self.processing_stage, task)

        # Record confidence for adaptive progression
        self.record_confidence(confidence)

        # Calculate collaborative context count
        collaborative_count = 0
        if hasattr(task, 'context_history') and task.context_history:
            collaborative_count = len(task.context_history)

        # Create result with prompt and settings metadata
        result = LLMResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            content=llm_response.content,
            position_info=position_info,
            llm_response=llm_response,
            processing_time=processing_time,
            timestamp=time.time(),
            confidence=confidence,
            processing_stage=self.processing_stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature_used=temperature,
            token_budget_allocated=effective_token_budget,
            collaborative_context_count=collaborative_count
        )
        
        # Record token usage with budget manager and log for monitoring
        if self.token_budget_manager:
            self.token_budget_manager.record_usage(self.agent_id, llm_response.tokens_used)

            # Debug logging for token usage monitoring
            budget_status = self.token_budget_manager.get_agent_status(self.agent_id)
            if budget_status:
                logger.debug(f"[TOKEN MONITOR] Agent {self.agent_id} ({self.agent_type}):")
                logger.debug(f"  - Stage budget: {stage_token_budget} tokens")
                logger.debug(f"  - Actual used: {llm_response.tokens_used} tokens")
                logger.debug(f"  - Total used: {budget_status['tokens_used']}/{budget_status['total_budget']} "
                           f"({budget_status['usage_ratio']:.1%})")
                logger.debug(f"  - Remaining: {budget_status['tokens_remaining']} tokens")

                # Warn if usage exceeds budget
                if llm_response.tokens_used > stage_token_budget:
                    logger.warning(f"[TOKEN OVERRUN] Agent {self.agent_id} used {llm_response.tokens_used} tokens "
                                 f"but was allocated only {stage_token_budget}")

        # Learn from token usage for future adaptive behavior (async)
        self._learn_from_token_usage(effective_token_budget, llm_response.tokens_used)

        # Record prompt metrics for optimization
        prompt_context = self._get_prompt_context(position_info.get("depth_ratio", 0.0))
        self._record_prompt_metrics(system_prompt, prompt_context, result)

        # Update statistics
        self.processing_results.append(result)
        self.total_tokens_used += llm_response.tokens_used
        self.total_processing_time += processing_time

        logger.info(f"Agent {self.agent_id} processed task {task.task_id} "
                   f"at depth {position_info.get('depth_ratio', 0):.2f} "
                   f"in {processing_time:.2f}s (async)")

        return result
        
    def process_task_with_llm(self, task: LLMTask, current_time: float,
                              central_post: Optional['CentralPost'] = None,
                              enable_streaming: bool = True) -> LLMResult:
        """
        Process task using LLM with position-aware prompting and optional streaming.

        Args:
            task: Task to process
            current_time: Current simulation time
            central_post: CentralPost for streaming updates (optional)
            enable_streaming: Use streaming if available (default True)

        Returns:
            LLM processing result
        """
        start_time = time.perf_counter()

        # Get position-aware prompts, token budget, and temperature
        system_prompt, stage_token_budget = self.create_position_aware_prompt(task, current_time)
        temperature = self.get_adaptive_temperature(current_time)
        position_info = self.get_position_info(current_time)

        # Apply learned adjustment factor to budget
        adjusted_budget = self.get_adjusted_token_budget(stage_token_budget)

        # Ensure adjusted budget doesn't exceed agent's max_tokens
        effective_token_budget = min(adjusted_budget, self.max_tokens)

        logger.debug(f"Agent {self.agent_id} token budget: base={stage_token_budget}, "
                    f"adjusted={adjusted_budget}, effective={effective_token_budget} "
                    f"(adjustment_factor={self._token_adjustment_factor:.2f})")

        # Build collaborative prompt if context history available
        if hasattr(task, 'context_history') and task.context_history:
            logger.info(f"Agent {self.agent_id} using collaborative context: "
                       f"{len(task.context_history)} previous outputs")
            user_prompt = self._build_collaborative_prompt(
                original_task=task.description,
                context_history=task.context_history,
                agent_type=self.agent_type
            )
        else:
            logger.info(f"Agent {self.agent_id} processing without collaborative context")
            user_prompt = task.description

        # Create token-aware stream controller for budget enforcement
        token_controller = TokenAwareStreamController(
            token_budget=effective_token_budget,
            soft_limit_ratio=0.85,  # Start concluding at 85% of budget
            conclusion_signal=""  # No explicit signal, let LLM finish naturally
        )

        # Streaming callback for real-time updates
        def streaming_callback(chunk):
            """Send partial thoughts to CentralPost during streaming."""
            if central_post and hasattr(central_post, 'receive_partial_thought'):
                try:
                    central_post.receive_partial_thought(
                        agent_id=self.agent_id,
                        partial_content=chunk.content,
                        accumulated=chunk.accumulated,
                        progress=self._progress,
                        metadata={
                            "agent_type": self.agent_type,
                            "checkpoint": self.get_current_checkpoint() if hasattr(self, 'get_current_checkpoint') else 0.0,
                            "tokens_so_far": chunk.tokens_so_far,
                            "position_info": position_info
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to send streaming chunk to hub: {e}")

        # Choose streaming or non-streaming based on feature flag
        # Note: Multi-server client pool requires async, so fall back to first available server
        if isinstance(self.llm_client, LMStudioClientPool):
            # For sync calls with pool, use the first available client
            server_name = self.llm_client.get_server_for_agent_type(self.agent_type)
            if server_name and server_name in self.llm_client.clients:
                client = self.llm_client.clients[server_name]

                # Use streaming if enabled and client supports it
                if enable_streaming and hasattr(client, 'complete_streaming'):
                    llm_response = client.complete_streaming(
                        agent_id=self.agent_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=effective_token_budget,
                        batch_interval=0.1,
                        callback=streaming_callback,
                        token_controller=token_controller
                    )
                else:
                    llm_response = client.complete(
                        agent_id=self.agent_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=effective_token_budget
                    )
            else:
                raise RuntimeError(f"No available server for agent type: {self.agent_type}")
        else:
            # Use streaming if enabled and client supports it
            if enable_streaming and hasattr(self.llm_client, 'complete_streaming'):
                llm_response = self.llm_client.complete_streaming(
                    agent_id=self.agent_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=effective_token_budget,
                    batch_interval=0.1,
                    callback=streaming_callback,
                    token_controller=token_controller
                )
            else:
                llm_response = self.llm_client.complete(
                    agent_id=self.agent_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=effective_token_budget
                )
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Increment processing stage
        self.processing_stage += 1

        # Calculate confidence based on position, content, and collaborative context
        confidence = self.calculate_confidence(current_time, llm_response.content, self.processing_stage, task)

        # Record confidence for adaptive progression
        self.record_confidence(confidence)

        # Calculate collaborative context count
        collaborative_count = 0
        if hasattr(task, 'context_history') and task.context_history:
            collaborative_count = len(task.context_history)

        # Create result with prompt and settings metadata
        result = LLMResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            content=llm_response.content,
            position_info=position_info,
            llm_response=llm_response,
            processing_time=processing_time,
            timestamp=time.time(),
            confidence=confidence,
            processing_stage=self.processing_stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature_used=temperature,
            token_budget_allocated=effective_token_budget,
            collaborative_context_count=collaborative_count
        )

        # Notify hub of streaming completion (if streaming was used)
        if enable_streaming and central_post and hasattr(central_post, 'finalize_streaming_thought'):
            try:
                central_post.finalize_streaming_thought(
                    agent_id=self.agent_id,
                    final_content=llm_response.content,
                    confidence=confidence
                )
            except Exception as e:
                logger.warning(f"Failed to finalize streaming thought with hub: {e}")

        # Record token usage with budget manager and log for monitoring
        if self.token_budget_manager:
            self.token_budget_manager.record_usage(self.agent_id, llm_response.tokens_used)

            # Debug logging for token usage monitoring
            budget_status = self.token_budget_manager.get_agent_status(self.agent_id)
            if budget_status:
                logger.debug(f"[TOKEN MONITOR] Agent {self.agent_id} ({self.agent_type}):")
                logger.debug(f"  - Stage budget: {stage_token_budget} tokens")
                logger.debug(f"  - Actual used: {llm_response.tokens_used} tokens")
                logger.debug(f"  - Total used: {budget_status['tokens_used']}/{budget_status['total_budget']} "
                           f"({budget_status['usage_ratio']:.1%})")
                logger.debug(f"  - Remaining: {budget_status['tokens_remaining']} tokens")

                # Warn if usage exceeds budget
                if llm_response.tokens_used > stage_token_budget:
                    logger.warning(f"[TOKEN OVERRUN] Agent {self.agent_id} used {llm_response.tokens_used} tokens "
                                 f"but was allocated only {stage_token_budget}")

        # Learn from token usage for future adaptive behavior
        self._learn_from_token_usage(effective_token_budget, llm_response.tokens_used)

        # Get efficiency metrics from token controller if streaming was used
        if enable_streaming and token_controller:
            controller_metrics = token_controller.get_metrics()
            logger.debug(f"Token controller metrics for {self.agent_id}: {controller_metrics}")
            # The controller's token count may differ slightly from LLM's reported count
            # Use LLM's count as authoritative for learning

        # Record prompt metrics for optimization
        prompt_context = self._get_prompt_context(position_info.get("depth_ratio", 0.0))
        self._record_prompt_metrics(system_prompt, prompt_context, result)

        # Update statistics
        self.processing_results.append(result)
        self.total_tokens_used += llm_response.tokens_used
        self.total_processing_time += processing_time

        logger.info(f"Agent {self.agent_id} processed task {task.task_id} "
                   f"at depth {position_info.get('depth_ratio', 0):.2f} "
                   f"in {processing_time:.2f}s")

        return result

    # Legacy method alias for backward compatibility
    async def process_task_async(self, task: LLMTask, current_time: float) -> LLMResult:
        """
        Asynchronously process task using LLM (legacy method).
        
        Args:
            task: Task to process
            current_time: Current simulation time
            
        Returns:
            LLM processing result
        """
        return await self.process_task_with_llm_async(task, current_time, RequestPriority.NORMAL)
    
    def share_result_to_central(self, result: LLMResult) -> Message:
        """
        Create message to share result with central post.

        Args:
            result: Processing result to share

        Returns:
            Message for central post communication
        """
        # Handle tokens_used for chunked vs non-chunked results
        tokens_used = 0
        if result.is_chunked and result.chunk_results:
            # Sum tokens from all chunks
            tokens_used = sum(chunk.metadata.get("tokens_used", 0) for chunk in result.chunk_results)
        elif result.llm_response:
            tokens_used = result.llm_response.tokens_used
        
        content_data = {
            "type": "AGENT_RESULT",
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "task_id": result.task_id,
            "content": result.content,  # Use FULL content for pattern detection (SYSTEM_ACTION_NEEDED, WEB_SEARCH_NEEDED)
            "full_content_available": result.full_content_available,
            "position_info": result.position_info,
            "tokens_used": tokens_used,
            "processing_time": result.processing_time,
            "confidence": result.confidence,
            "processing_stage": result.processing_stage,
            "summary": self._create_result_summary(result)
        }
        
        # Add chunking metadata if applicable
        if result.is_chunked:
            content_data["chunking_metadata"] = result.get_chunking_metadata()
            content_data["chunking_strategy"] = result.chunking_strategy

            # For streaming synthesis, make chunks available to synthesis agents
            if result.chunking_strategy == "streaming" and result.chunk_results:
                content_data["streaming_chunks"] = [
                    {
                        "chunk_index": chunk.chunk_index,
                        "aspect": chunk.metadata.get("aspect", f"Section {chunk.chunk_index + 1}"),
                        "content": chunk.content_chunk,
                        "is_final": chunk.is_final,
                        "timestamp": chunk.timestamp
                    }
                    for chunk in result.chunk_results
                ]

        # Add validation metadata for critic agents (optional, backward compatible)
        if self.agent_type == 'critic':
            try:
                from src.workflows.truth_assessment import calculate_validation_score, get_validation_flags

                validation_score = calculate_validation_score(
                    content={"result": result.content},
                    source_agent=self.agent_id,
                    domain="workflow_task",
                    confidence_level="HIGH" if result.confidence > 0.8 else "MEDIUM"
                )
                validation_flags = get_validation_flags(
                    content={"result": result.content},
                    source_agent=self.agent_id,
                    domain="workflow_task"
                )

                content_data['validation_score'] = validation_score
                content_data['validation_flags'] = validation_flags
            except Exception as e:
                logger.warning(f"Could not add validation metadata for critic {self.agent_id}: {e}")

        return Message(
            sender_id=self.agent_id,
            message_type=MessageType.STATUS_UPDATE,
            content=content_data,
            timestamp=result.timestamp
        )
    
    def _create_result_summary(self, result: LLMResult) -> str:
        """Create concise summary of processing result."""
        content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
        depth = result.position_info.get("depth_ratio", 0.0)
        
        return f"[{self.agent_type.upper()} @ depth {depth:.2f}] {content_preview}"
    
    def receive_shared_context(self, message: Dict[str, Any]) -> None:
        """
        Receive and store shared context from other agents.

        Args:
            message: Shared context message
        """
        self.received_messages.append(message)

        # Extract relevant context
        if message.get("type") == "AGENT_RESULT":
            key = f"{message.get('agent_type', 'unknown')}_{message.get('agent_id', '')}"
            self.shared_context[key] = message.get("summary", "")

    def process_synthesis_feedback(self, feedback_message: Dict[str, Any]) -> None:
        """
        Process feedback from CentralPost about synthesis and contribution quality.

        This implements the agent-side Feedback Integration Protocol, enabling
        learning and adaptation based on how contributions were used in synthesis.

        Args:
            feedback_message: Message containing feedback from CentralPost
                - For SYNTHESIS_FEEDBACK: synthesis results and quality
                - For CONTRIBUTION_EVALUATION: usefulness score and calibration data
        """
        from src.communication.message_types import MessageType

        message_type_str = feedback_message.get('type', '')

        if message_type_str == MessageType.SYNTHESIS_FEEDBACK.value:
            # Process synthesis feedback
            synthesis_confidence = feedback_message.get('synthesis_confidence', 0.0)
            agents_synthesized = feedback_message.get('agents_synthesized', 0)

            logger.info(f"📥 {self.agent_id} received synthesis feedback: "
                       f"confidence={synthesis_confidence:.2f}, agents={agents_synthesized}")

        elif message_type_str == MessageType.CONTRIBUTION_EVALUATION.value:
            # Process contribution evaluation - this is the critical learning signal
            usefulness_score = feedback_message.get('usefulness_score', 0.0)
            incorporated = feedback_message.get('incorporated_in_synthesis', False)
            confidence_calibration = feedback_message.get('confidence_calibration', 0.0)
            calibration_quality = feedback_message.get('calibration_quality', 'unknown')

            # Record contribution in history
            self.contribution_history.append({
                'timestamp': time.time(),
                'usefulness_score': usefulness_score,
                'incorporated': incorporated,
                'confidence_calibration': confidence_calibration,
                'calibration_quality': calibration_quality
            })

            # Update running average of synthesis integration rate
            self._total_feedback_received += 1
            old_rate = self.synthesis_integration_rate
            self.synthesis_integration_rate = (
                (old_rate * (self._total_feedback_received - 1) + usefulness_score) /
                self._total_feedback_received
            )

            # Log learning insights
            logger.info(f"📊 {self.agent_id} contribution evaluation: "
                       f"usefulness={usefulness_score:.2f}, "
                       f"incorporated={'YES' if incorporated else 'NO'}, "
                       f"calibration={confidence_calibration:+.2f} ({calibration_quality})")

            logger.info(f"   Integration rate: {old_rate:.2f} → {self.synthesis_integration_rate:.2f}")

            # Adapt behavior based on feedback
            self._adapt_based_on_feedback(usefulness_score, confidence_calibration, calibration_quality)

    def _adapt_based_on_feedback(self, usefulness_score: float,
                                 confidence_calibration: float,
                                 calibration_quality: str) -> None:
        """
        Adapt agent behavior based on synthesis feedback.

        Args:
            usefulness_score: How useful the contribution was (0.0-1.0)
            confidence_calibration: Synthesis confidence - agent confidence
            calibration_quality: 'good' or 'needs_adjustment'
        """
        # Adjust future confidence calculations if calibration is poor
        if calibration_quality == 'needs_adjustment':
            if confidence_calibration > 0:
                # Agent was under-confident, boost future confidence slightly
                logger.debug(f"   {self.agent_id}: Under-confident by {abs(confidence_calibration):.2f}, "
                           "will be slightly bolder")
            else:
                # Agent was over-confident, reduce future confidence
                logger.debug(f"   {self.agent_id}: Over-confident by {abs(confidence_calibration):.2f}, "
                           "will be more cautious")

        # If consistently low usefulness, agent might need to adjust approach
        if len(self.contribution_history) >= 3:
            recent_usefulness = [c['usefulness_score'] for c in self.contribution_history[-3:]]
            avg_recent = sum(recent_usefulness) / len(recent_usefulness)

            if avg_recent < 0.3:
                logger.warning(f"⚠️  {self.agent_id}: Low recent usefulness ({avg_recent:.2f}), "
                             "may need to adjust approach")
            elif avg_recent > 0.7:
                logger.info(f"✨ {self.agent_id}: High recent usefulness ({avg_recent:.2f}), "
                          "current approach working well!")

    def influence_agent_behavior(self, other_agent: "LLMAgent", influence_type: str, strength: float) -> None:
        """
        Influence another agent's behavior based on collaboration.
        
        Args:
            other_agent: Agent to influence
            influence_type: Type of influence ('accelerate', 'slow', 'pause', 'redirect')
            strength: Influence strength (0.0 to 1.0)
        """
        if strength <= 0.0 or other_agent.agent_id == self.agent_id:
            return  # No influence or self-influence
        
        # Record the influence relationship
        if other_agent.agent_id not in self.influence_strength:
            self.influence_strength[other_agent.agent_id] = 0.0
        self.influence_strength[other_agent.agent_id] += strength * 0.1  # Cumulative influence
        
        if self.agent_id not in other_agent.influenced_by:
            other_agent.influenced_by.append(self.agent_id)
        
        # Apply influence based on type and agent compatibility
        compatibility = self._calculate_agent_compatibility(other_agent)
        effective_strength = strength * compatibility
        
        if influence_type == "accelerate" and effective_strength > 0.3:
            # Speed up the other agent if they're compatible
            current_velocity = other_agent.velocity
            other_agent.set_velocity_multiplier(min(current_velocity * 1.2, 2.0))
        
        elif influence_type == "slow" and effective_strength > 0.4:
            # Slow down if there's strong incompatibility
            current_velocity = other_agent.velocity
            other_agent.set_velocity_multiplier(max(current_velocity * 0.8, 0.3))
        
        elif influence_type == "pause" and effective_strength > 0.6:
            # Pause for consideration of conflicting approaches
            other_agent.pause_for_duration(0.1 * effective_strength, 0.0)  # Brief pause
        
        # Record collaboration
        self.collaboration_history.append({
            "timestamp": time.time(),
            "other_agent": other_agent.agent_id,
            "influence_type": influence_type,
            "strength": strength,
            "effective_strength": effective_strength,
            "compatibility": compatibility
        })
    
    def _calculate_agent_compatibility(self, other_agent: "LLMAgent") -> float:
        """
        Calculate compatibility between this agent and another.
        
        Args:
            other_agent: Other agent to assess compatibility with
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Type compatibility matrix
        type_compatibility = {
            ("research", "research"): 0.8,     # Research agents collaborate well
            ("research", "analysis"): 0.9,    # Research feeds analysis
            ("research", "synthesis"): 0.7,   # Research provides raw material
            ("research", "critic"): 0.6,      # Some tension but productive
            
            ("analysis", "analysis"): 0.7,    # Analysis agents can complement
            ("analysis", "synthesis"): 0.9,   # Analysis feeds synthesis
            ("analysis", "critic"): 0.8,      # Analysis benefits from critique
            
            ("synthesis", "synthesis"): 0.5,  # May compete for final output
            ("synthesis", "critic"): 0.8,     # Synthesis benefits from review
            
            ("critic", "critic"): 0.6,        # Critics can disagree
        }
        
        # Get base compatibility from types
        type_pair = (self.agent_type, other_agent.agent_type)
        reverse_type_pair = (other_agent.agent_type, self.agent_type)
        
        base_compatibility = type_compatibility.get(
            type_pair, type_compatibility.get(reverse_type_pair, 0.5)
        )
        
        # Modify based on confidence histories
        if (len(self._confidence_history) > 2 and 
            len(other_agent._confidence_history) > 2):
            
            my_trend = self._confidence_history[-1] - self._confidence_history[-2]
            their_trend = other_agent._confidence_history[-1] - other_agent._confidence_history[-2]
            
            # Agents with similar confidence trends are more compatible
            trend_similarity = 1.0 - abs(my_trend - their_trend)
            base_compatibility = (base_compatibility + trend_similarity) / 2
        
        return max(0.0, min(base_compatibility, 1.0))
    
    def assess_collaboration_opportunities(self, available_agents: List["LLMAgent"], 
                                         current_time: float) -> List[Dict[str, Any]]:
        """
        Assess opportunities for collaboration with other agents.
        
        Args:
            available_agents: List of other agents available for collaboration
            current_time: Current simulation time
            
        Returns:
            List of collaboration opportunities with recommendations
        """
        opportunities = []
        
        for other_agent in available_agents:
            if other_agent.agent_id == self.agent_id or other_agent.state != AgentState.ACTIVE:
                continue
            
            compatibility = self._calculate_agent_compatibility(other_agent)
            
            # Skip if compatibility is too low
            if compatibility < 0.3:
                continue
            
            # Assess potential collaboration based on current states
            opportunity = {
                "agent_id": other_agent.agent_id,
                "agent_type": other_agent.agent_type,
                "compatibility": compatibility,
                "recommended_influence": self._recommend_influence_type(other_agent),
                "confidence": other_agent._confidence_history[-1] if other_agent._confidence_history else 0.5,
                "distance": abs(self._progress - other_agent._progress)
            }
            
            opportunities.append(opportunity)
        
        # Sort by potential value (compatibility * confidence, adjusted for distance)
        opportunities.sort(key=lambda x: x["compatibility"] * x["confidence"] * (1.0 - x["distance"] * 0.5), reverse=True)
        
        return opportunities
    
    def _recommend_influence_type(self, other_agent: "LLMAgent") -> str:
        """
        Recommend type of influence to apply to another agent.
        
        Args:
            other_agent: Agent to recommend influence for
            
        Returns:
            Recommended influence type
        """
        if not other_agent._confidence_history:
            return "accelerate"  # Default to acceleration for new agents
        
        other_confidence = other_agent._confidence_history[-1]
        my_confidence = self._confidence_history[-1] if self._confidence_history else 0.5
        
        # If other agent has low confidence and I have high confidence, accelerate them
        if other_confidence < 0.5 and my_confidence > 0.7:
            return "accelerate"
        
        # If other agent has much higher confidence, slow down to learn from them
        elif other_confidence > my_confidence + 0.3:
            return "slow"
        
        # If confidence gap is large and we're incompatible, suggest pause
        elif abs(other_confidence - my_confidence) > 0.4:
            return "pause"
        
        # Default to acceleration for collaborative growth
        return "accelerate"
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics including emergent behavior metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "spawn_time": self.spawn_time,
            "progress": self._progress,
            "total_tasks_processed": len(self.processing_results),
            "total_tokens_used": self.total_tokens_used,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": (self.total_processing_time / len(self.processing_results)
                                      if self.processing_results else 0.0),
            "messages_received": len(self.received_messages),
            "shared_context_items": len(self.shared_context),
            
            # Emergent behavior metrics
            "influenced_by_count": len(self.influenced_by),
            "influences_given": len(self.influence_strength),
            "total_influence_received": sum(self.influence_strength.values()),
            "collaboration_count": len(self.collaboration_history),
            "velocity": self.velocity,
            "confidence_history": self._confidence_history.copy(),
            "progression_info": self.get_progression_info()
        }
        
        # Add token budget information if available
        if self.token_budget_manager:
            budget_status = self.token_budget_manager.get_agent_status(self.agent_id)
            if budget_status:
                stats["token_budget"] = budget_status
        
        return stats


    # ========================================
    # System Action Methods
    # ========================================

    def request_action(self, command: str, context: str = "") -> str:
        """
        Request system action execution through CentralPost.

        This method allows agents to request command execution with automatic
        trust classification and approval workflow. Commands are routed through
        CentralPost which handles trust management and execution.

        Args:
            command: System command to execute
            context: Context/reason for command execution

        Returns:
            Action ID for tracking the request

        Example:
            action_id = agent.request_action(
                command="pip list",
                context="Checking installed packages for task analysis"
            )
        """
        if not hasattr(self, 'spoke') or self.spoke is None:
            logger.warning(f"Agent {self.agent_id} has no spoke connection, cannot request action")
            return ""

        # Create system action request message
        message = Message(
            sender_id=self.agent_id,
            message_type=MessageType.SYSTEM_ACTION_REQUEST,
            content={
                "command": command,
                "context": context,
                "agent_type": self.agent_type,
                "position": self._progress
            },
            timestamp=time.time()
        )

        # Send message through spoke to CentralPost
        self.spoke.send_message(message)

        logger.info(f"Agent {self.agent_id} requested system action: {command[:50]}")

        # Return a temporary action_id (will be replaced by CentralPost's actual ID)
        # In practice, agents should wait for SYSTEM_ACTION_RESULT message
        return f"pending_{self.agent_id}_{int(time.time() * 1000)}"

    def check_action_result(self, action_id: str) -> Optional['CommandResult']:
        """
        Check if system action has completed and retrieve result.

        This method polls for action results by checking received messages
        for SYSTEM_ACTION_RESULT messages matching the action_id.

        Args:
            action_id: Action ID returned by request_action()

        Returns:
            CommandResult if action completed, None if still pending

        Example:
            result = agent.check_action_result(action_id)
            if result:
                print(f"Command output: {result.stdout}")
        """
        # Check received messages for action result
        for message in self.received_messages:
            if message.get("type") == "SYSTEM_ACTION_RESULT":
                if message.get("action_id") == action_id:
                    # Extract CommandResult from message
                    from src.execution import CommandResult, ErrorCategory

                    result_data = message.get("result", {})

                    # Reconstruct CommandResult
                    error_category = None
                    if result_data.get("error_category"):
                        error_category = ErrorCategory(result_data["error_category"])

                    result = CommandResult(
                        command=result_data.get("command", ""),
                        exit_code=result_data.get("exit_code", -1),
                        stdout=result_data.get("stdout", ""),
                        stderr=result_data.get("stderr", ""),
                        duration=result_data.get("duration", 0.0),
                        success=result_data.get("success", False),
                        error_category=error_category,
                        cwd=result_data.get("cwd", ""),
                        venv_active=result_data.get("venv_active", False),
                        output_size=result_data.get("output_size", 0),
                        timestamp=result_data.get("timestamp", time.time())
                    )

                    return result

        return None

    def wait_for_action_result(self, action_id: str, timeout: float = 10.0) -> Optional['CommandResult']:
        """
        Wait for system action to complete with timeout.

        Blocks until action completes or timeout expires.

        Args:
            action_id: Action ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            CommandResult if action completed, None if timeout

        Example:
            result = agent.wait_for_action_result(action_id, timeout=30.0)
            if result:
                print(f"Success: {result.success}")
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.check_action_result(action_id)
            if result is not None:
                return result

            # Brief sleep to avoid busy waiting
            time.sleep(0.1)

        logger.warning(f"Agent {self.agent_id} timed out waiting for action {action_id}")
        return None

    def get_system_state(self) -> Dict[str, Any]:
        """
        Get current system state from CentralPost.

        Retrieves information about current working directory, virtual environment
        status, and other system details.

        Returns:
            Dictionary with system state details:
                - cwd: Current working directory
                - venv_active: Whether virtual environment is active
                - venv_path: Path to active virtual environment
                - user: Current user
                - home: Home directory
                - python_executable: Path to Python executable

        Example:
            state = agent.get_system_state()
            if not state.get('venv_active'):
                agent.request_action("source .venv/bin/activate")
        """
        if not hasattr(self, 'spoke') or self.spoke is None:
            logger.warning(f"Agent {self.agent_id} has no spoke connection")
            return {}

        # For now, return cached system state if available
        # In future, could send a message to CentralPost to get fresh state
        if hasattr(self, '_cached_system_state'):
            return self._cached_system_state

        # Default system state
        return {
            'cwd': os.getcwd() if 'os' in dir() else '.',
            'venv_active': False,
            'venv_path': None,
            'user': 'unknown',
            'home': '~',
            'python_executable': '/usr/bin/python3'
        }

    def can_execute_commands(self) -> bool:
        """
        Check if agent has capability to request system command execution.

        Returns:
            True if agent can request commands, False otherwise
        """
        # Agent can execute commands if it has a spoke connection to CentralPost
        return hasattr(self, 'spoke') and self.spoke is not None

    def request_venv_activation(self, venv_path: Optional[str] = None) -> str:
        """
        Request activation of virtual environment.

        Convenience method for common venv activation operation.

        Args:
            venv_path: Path to venv (None = auto-detect)

        Returns:
            Action ID for tracking

        Example:
            action_id = agent.request_venv_activation()
            result = agent.wait_for_action_result(action_id)
        """
        if venv_path:
            command = f"source {venv_path}/bin/activate"
            context = f"Activating virtual environment at {venv_path}"
        else:
            # Auto-detect common venv locations
            command = "source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || source env/bin/activate"
            context = "Auto-detecting and activating virtual environment"

        return self.request_action(command, context)

    def request_package_install(self, package: str, use_pip: bool = True) -> str:
        """
        Request installation of Python package.

        Convenience method for package installation.

        Args:
            package: Package name to install
            use_pip: Use pip (True) or conda (False)

        Returns:
            Action ID for tracking

        Example:
            action_id = agent.request_package_install("numpy")
        """
        if use_pip:
            command = f"pip install {package}"
            context = f"Installing Python package: {package}"
        else:
            command = f"conda install -y {package}"
            context = f"Installing package via conda: {package}"

        return self.request_action(command, context)

    def request_file_check(self, filepath: str) -> str:
        """
        Request check if file exists.

        Convenience method for file existence checking.

        Args:
            filepath: Path to file

        Returns:
            Action ID for tracking

        Example:
            action_id = agent.request_file_check("requirements.txt")
            result = agent.wait_for_action_result(action_id)
            if result and result.success:
                print("File exists!")
        """
        command = f"test -f {filepath} && echo 'exists' || echo 'not found'"
        context = f"Checking if file exists: {filepath}"

        return self.request_action(command, context)
def create_llm_agents(helix: HelixGeometry, llm_client: LMStudioClient,
                     agent_configs: List[Dict[str, Any]]) -> List[LLMAgent]:
    """
    Create multiple LLM agents with specified configurations.
    
    Args:
        helix: Helix geometry for agent paths
        llm_client: LM Studio client for LLM inference
        agent_configs: List of agent configuration dictionaries
        
    Returns:
        List of configured LLM agents
    """
    agents = []
    
    for config in agent_configs:
        agent = LLMAgent(
            agent_id=config["agent_id"],
            spawn_time=config["spawn_time"],
            helix=helix,
            llm_client=llm_client,
            agent_type=config.get("agent_type", "general"),
            temperature_range=config.get("temperature_range", (0.1, 0.9))
        )
        agents.append(agent)
    
    return agents


def create_specialized_agent_configs(num_research: int = 3, num_analysis: int = 2,
                                   num_synthesis: int = 1, random_seed: int = 42069) -> List[Dict[str, Any]]:
    """
    Create agent configurations for typical Felix multi-agent task.
    
    Args:
        num_research: Number of research agents (spawn early)
        num_analysis: Number of analysis agents (spawn mid)
        num_synthesis: Number of synthesis agents (spawn late)
        random_seed: Random seed for spawn timing
        
    Returns:
        List of agent configuration dictionaries
    """
    import random
    random.seed(random_seed)
    
    configs = []
    agent_id = 0
    
    # Research agents - spawn early (0.0-0.4)
    for i in range(num_research):
        configs.append({
            "agent_id": f"research_{agent_id:03d}",
            "spawn_time": random.uniform(0.0, 0.4),
            "agent_type": "research",
            "temperature_range": (0.3, 0.9)
        })
        agent_id += 1
    
    # Analysis agents - spawn mid (0.3-0.7)
    for i in range(num_analysis):
        configs.append({
            "agent_id": f"analysis_{agent_id:03d}",
            "spawn_time": random.uniform(0.3, 0.7),
            "agent_type": "analysis", 
            "temperature_range": (0.2, 0.7)
        })
        agent_id += 1
    
    # Synthesis agents - spawn late (0.6-1.0)
    for i in range(num_synthesis):
        configs.append({
            "agent_id": f"synthesis_{agent_id:03d}",
            "spawn_time": random.uniform(0.6, 1.0),
            "agent_type": "synthesis",
            "temperature_range": (0.1, 0.5)
        })
        agent_id += 1
    
    return configs
