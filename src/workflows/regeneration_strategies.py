"""
Regeneration Strategies for Low-Confidence Synthesis.

Provides multiple strategies for improving synthesis quality when initial
synthesis confidence is below threshold. Users can choose how to regenerate:
- Parameter adjustment (lower temp, more tokens)
- Context injection (user provides clarification)
- Spawn more agents (fresh research/critics)
- Web search boost (force external search)
- Knowledge expansion (lower retrieval threshold)

This module addresses the human-in-the-loop gap for synthesis decisions,
giving users meaningful choices rather than binary accept/reject.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)


class RegenerationStrategy(Enum):
    """Strategies for regenerating low-confidence synthesis."""

    PARAMETER_ADJUST = "parameter_adjust"
    """Adjust LLM parameters: lower temperature, increase token budget."""

    CONTEXT_INJECTION = "context_injection"
    """User provides additional context/clarification to improve synthesis."""

    SPAWN_MORE_AGENTS = "spawn_more_agents"
    """Force spawn additional research and critic agents."""

    WEB_SEARCH_BOOST = "web_search_boost"
    """Force web search regardless of confidence threshold."""

    KNOWLEDGE_EXPAND = "knowledge_expand"
    """Lower similarity threshold to retrieve more knowledge context."""


@dataclass
class RegenerationRequest:
    """Request for synthesis regeneration."""

    strategy: RegenerationStrategy
    """The regeneration strategy to execute."""

    user_context: Optional[str] = None
    """Additional context from user (for CONTEXT_INJECTION strategy)."""

    parameter_overrides: Optional[Dict[str, Any]] = None
    """Custom parameter overrides (for PARAMETER_ADJUST strategy)."""

    max_attempts: int = 1
    """Maximum regeneration attempts for this request."""

    request_id: str = field(default_factory=lambda: f"regen_{int(time.time()*1000)}")
    """Unique identifier for this regeneration request."""


@dataclass
class RegenerationResult:
    """Result of a regeneration attempt."""

    success: bool
    """Whether regeneration succeeded."""

    strategy_used: RegenerationStrategy
    """The strategy that was executed."""

    new_synthesis: Optional[Dict[str, Any]] = None
    """New synthesis result if successful."""

    confidence_delta: float = 0.0
    """Change in confidence from original synthesis."""

    execution_time: float = 0.0
    """Time taken for regeneration in seconds."""

    message: str = ""
    """Human-readable status message."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional details about the regeneration."""


class RegenerationExecutor:
    """
    Executes regeneration strategies for low-confidence synthesis.

    Coordinates between synthesis engine, agent factory, knowledge retriever,
    and web search to implement various regeneration approaches.
    """

    def __init__(
        self,
        synthesis_engine: Any,
        central_post: Optional[Any] = None,
        agent_factory: Optional[Any] = None,
        knowledge_retriever: Optional[Any] = None,
        web_search_coordinator: Optional[Any] = None
    ):
        """
        Initialize the regeneration executor.

        Args:
            synthesis_engine: SynthesisEngine instance for re-synthesis
            central_post: Optional CentralPost for message coordination
            agent_factory: Optional AgentFactory for spawning new agents
            knowledge_retriever: Optional knowledge retrieval interface
            web_search_coordinator: Optional web search coordinator
        """
        self.synthesis_engine = synthesis_engine
        self.central_post = central_post
        self.agent_factory = agent_factory
        self.knowledge_retriever = knowledge_retriever
        self.web_search = web_search_coordinator

        logger.info("RegenerationExecutor initialized")

    def execute(
        self,
        request: RegenerationRequest,
        original_task: str,
        original_context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> RegenerationResult:
        """
        Execute a regeneration strategy and return new synthesis.

        Args:
            request: The regeneration request with strategy and parameters
            original_task: Original task description
            original_context: Context from original synthesis attempt
            progress_callback: Optional callback for progress updates

        Returns:
            RegenerationResult with new synthesis or error details
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(f"Starting {request.strategy.value} regeneration...")

        try:
            if request.strategy == RegenerationStrategy.PARAMETER_ADJUST:
                result = self._execute_parameter_adjust(
                    request, original_task, original_context, progress_callback
                )
            elif request.strategy == RegenerationStrategy.CONTEXT_INJECTION:
                result = self._execute_context_injection(
                    request, original_task, original_context, progress_callback
                )
            elif request.strategy == RegenerationStrategy.SPAWN_MORE_AGENTS:
                result = self._execute_spawn_more_agents(
                    request, original_task, original_context, progress_callback
                )
            elif request.strategy == RegenerationStrategy.WEB_SEARCH_BOOST:
                result = self._execute_web_search_boost(
                    request, original_task, original_context, progress_callback
                )
            elif request.strategy == RegenerationStrategy.KNOWLEDGE_EXPAND:
                result = self._execute_knowledge_expand(
                    request, original_task, original_context, progress_callback
                )
            else:
                result = RegenerationResult(
                    success=False,
                    strategy_used=request.strategy,
                    message=f"Unknown strategy: {request.strategy.value}"
                )

            result.execution_time = time.time() - start_time

            # Calculate confidence delta if we have both results
            if result.new_synthesis and 'confidence' in result.new_synthesis:
                original_conf = original_context.get('confidence', 0.0)
                new_conf = result.new_synthesis.get('confidence', 0.0)
                result.confidence_delta = new_conf - original_conf

            logger.info(
                f"Regeneration complete: strategy={request.strategy.value}, "
                f"success={result.success}, delta={result.confidence_delta:+.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                execution_time=time.time() - start_time,
                message=f"Regeneration failed: {str(e)}"
            )

    def _execute_parameter_adjust(
        self,
        request: RegenerationRequest,
        original_task: str,
        original_context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]]
    ) -> RegenerationResult:
        """
        Regenerate with adjusted parameters: lower temperature, more tokens.

        This strategy produces more focused, deterministic output.
        """
        if progress_callback:
            progress_callback("Adjusting synthesis parameters for focused output...")

        # Get original parameters
        original_temp = original_context.get('temperature', 0.4)
        original_tokens = original_context.get('max_tokens', 2000)

        # Apply adjustments (or use overrides)
        overrides = request.parameter_overrides or {}
        new_temp = overrides.get('temperature', max(0.1, original_temp * 0.5))
        new_tokens = overrides.get('max_tokens', int(original_tokens * 1.5))

        logger.info(f"Parameter adjust: temp {original_temp:.2f}->{new_temp:.2f}, "
                   f"tokens {original_tokens}->{new_tokens}")

        # Re-run synthesis with adjusted parameters
        # Note: This requires the synthesis engine to accept parameter overrides
        # For now, we'll store the adjusted params in context for the caller to use
        return RegenerationResult(
            success=True,
            strategy_used=request.strategy,
            message=f"Parameters adjusted: temp={new_temp:.2f}, max_tokens={new_tokens}",
            details={
                'adjusted_temperature': new_temp,
                'adjusted_max_tokens': new_tokens,
                'original_temperature': original_temp,
                'original_max_tokens': original_tokens,
                'requires_resynthesize': True
            }
        )

    def _execute_context_injection(
        self,
        request: RegenerationRequest,
        original_task: str,
        original_context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]]
    ) -> RegenerationResult:
        """
        Regenerate with user-provided additional context.

        The user's clarification is appended to the original task.
        """
        if not request.user_context:
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message="No user context provided for context injection"
            )

        if progress_callback:
            progress_callback("Injecting user context into task...")

        # Build enriched task with user context
        enriched_task = f"""{original_task}

---
Additional context from user:
{request.user_context}
---"""

        logger.info(f"Context injected: +{len(request.user_context)} chars")

        return RegenerationResult(
            success=True,
            strategy_used=request.strategy,
            message="User context injected into task",
            details={
                'enriched_task': enriched_task,
                'context_length': len(request.user_context),
                'requires_resynthesize': True
            }
        )

    def _execute_spawn_more_agents(
        self,
        request: RegenerationRequest,
        original_task: str,
        original_context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]]
    ) -> RegenerationResult:
        """
        Spawn additional research and critic agents for more perspectives.

        Requires agent_factory and central_post to be configured.
        """
        if not self.agent_factory:
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message="Agent factory not available for spawning"
            )

        if progress_callback:
            progress_callback("Spawning additional research and critic agents...")

        spawned_agents = []

        try:
            # Try to spawn a research agent
            if hasattr(self.agent_factory, 'create_research_agent'):
                research = self.agent_factory.create_research_agent(
                    helix=getattr(self.central_post, 'helix', None),
                    llm_client=getattr(self.central_post, 'llm_client', None),
                    spawn_time=time.time(),
                    focus='deep_research'
                )
                spawned_agents.append({
                    'agent_id': research.agent_id,
                    'type': 'research',
                    'purpose': 'Additional research perspective'
                })

                # Register with central post if available
                if self.central_post and hasattr(self.central_post, 'register_agent'):
                    self.central_post.register_agent(research.agent_id, {
                        'type': 'research',
                        'spawned_for': 'regeneration'
                    })

            # Try to spawn a critic agent
            if hasattr(self.agent_factory, 'create_critic_agent'):
                critic = self.agent_factory.create_critic_agent(
                    helix=getattr(self.central_post, 'helix', None),
                    llm_client=getattr(self.central_post, 'llm_client', None),
                    spawn_time=time.time(),
                    focus='validation'
                )
                spawned_agents.append({
                    'agent_id': critic.agent_id,
                    'type': 'critic',
                    'purpose': 'Additional validation'
                })

                if self.central_post and hasattr(self.central_post, 'register_agent'):
                    self.central_post.register_agent(critic.agent_id, {
                        'type': 'critic',
                        'spawned_for': 'regeneration'
                    })

            logger.info(f"Spawned {len(spawned_agents)} agents for regeneration")

            return RegenerationResult(
                success=len(spawned_agents) > 0,
                strategy_used=request.strategy,
                message=f"Spawned {len(spawned_agents)} additional agents",
                details={
                    'spawned_agents': spawned_agents,
                    'requires_agent_execution': True,
                    'requires_resynthesize': True
                }
            )

        except Exception as e:
            logger.error(f"Failed to spawn agents: {e}")
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message=f"Failed to spawn agents: {str(e)}"
            )

    def _execute_web_search_boost(
        self,
        request: RegenerationRequest,
        original_task: str,
        original_context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]]
    ) -> RegenerationResult:
        """
        Force web search regardless of normal confidence thresholds.

        Brings in fresh external information to improve synthesis.
        """
        if not self.web_search:
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message="Web search coordinator not available"
            )

        if progress_callback:
            progress_callback("Performing forced web search for additional context...")

        try:
            # Formulate search query from task
            if hasattr(self.web_search, 'formulate_query'):
                query = self.web_search.formulate_query(original_task, [])
            else:
                # Simple fallback: use task as query
                query = original_task[:200]  # Limit query length

            # Perform search with force flag
            if hasattr(self.web_search, 'perform_search'):
                search_results = self.web_search.perform_search(
                    query=query,
                    force=True
                )

                result_count = len(search_results) if isinstance(search_results, list) else 0
                logger.info(f"Web search returned {result_count} results")

                return RegenerationResult(
                    success=True,
                    strategy_used=request.strategy,
                    message=f"Web search completed: {result_count} results",
                    details={
                        'search_query': query,
                        'result_count': result_count,
                        'search_results': search_results,
                        'requires_resynthesize': True
                    }
                )
            else:
                return RegenerationResult(
                    success=False,
                    strategy_used=request.strategy,
                    message="Web search coordinator missing perform_search method"
                )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message=f"Web search failed: {str(e)}"
            )

    def _execute_knowledge_expand(
        self,
        request: RegenerationRequest,
        original_task: str,
        original_context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]]
    ) -> RegenerationResult:
        """
        Lower similarity threshold to retrieve more knowledge context.

        Casts a wider net in the knowledge base for relevant information.
        """
        if not self.knowledge_retriever:
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message="Knowledge retriever not available"
            )

        if progress_callback:
            progress_callback("Expanding knowledge search with lower threshold...")

        try:
            # Use lower threshold than default (typically 0.75 -> 0.5)
            expanded_threshold = 0.5
            expanded_max_results = 25  # More than typical default

            if hasattr(self.knowledge_retriever, 'search'):
                expanded_knowledge = self.knowledge_retriever.search(
                    query=original_task,
                    similarity_threshold=expanded_threshold,
                    max_results=expanded_max_results
                )

                result_count = len(expanded_knowledge) if isinstance(expanded_knowledge, list) else 0
                logger.info(f"Expanded knowledge search returned {result_count} results")

                return RegenerationResult(
                    success=True,
                    strategy_used=request.strategy,
                    message=f"Knowledge search expanded: {result_count} results",
                    details={
                        'similarity_threshold': expanded_threshold,
                        'max_results': expanded_max_results,
                        'result_count': result_count,
                        'expanded_knowledge': expanded_knowledge,
                        'requires_resynthesize': True
                    }
                )
            else:
                return RegenerationResult(
                    success=False,
                    strategy_used=request.strategy,
                    message="Knowledge retriever missing search method"
                )

        except Exception as e:
            logger.error(f"Knowledge expansion failed: {e}")
            return RegenerationResult(
                success=False,
                strategy_used=request.strategy,
                message=f"Knowledge expansion failed: {str(e)}"
            )

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Get list of available regeneration strategies with metadata.

        Returns:
            List of strategy descriptions for UI display
        """
        strategies = [
            {
                'id': RegenerationStrategy.PARAMETER_ADJUST.value,
                'label': 'Regenerate (more focused)',
                'description': 'Lower temperature, more tokens for deterministic output',
                'requires_input': False,
                'available': True
            },
            {
                'id': RegenerationStrategy.CONTEXT_INJECTION.value,
                'label': 'Add context and regenerate',
                'description': 'Provide additional clarification to improve results',
                'requires_input': True,
                'input_prompt': 'Provide additional context to improve the synthesis:',
                'available': True
            },
            {
                'id': RegenerationStrategy.SPAWN_MORE_AGENTS.value,
                'label': 'Spawn more agents',
                'description': 'Add research and critic agents for more perspectives',
                'requires_input': False,
                'available': self.agent_factory is not None
            },
            {
                'id': RegenerationStrategy.WEB_SEARCH_BOOST.value,
                'label': 'Search web and regenerate',
                'description': 'Force web search for fresh external information',
                'requires_input': False,
                'available': self.web_search is not None
            },
            {
                'id': RegenerationStrategy.KNOWLEDGE_EXPAND.value,
                'label': 'Expand knowledge search',
                'description': 'Lower threshold to retrieve more context from knowledge base',
                'requires_input': False,
                'available': self.knowledge_retriever is not None
            }
        ]

        return strategies
