"""
Felix Unified Entry Point

This module provides the single entry point for all Felix interactions.
Whether Simple mode or Workflow mode, all requests go through run_felix().

Usage:
    from src.workflows.felix_inference import run_felix

    # Simple mode (direct Felix inference)
    result = run_felix(felix_system, "Hello!", mode="direct")

    # Workflow mode (full multi-agent orchestration)
    result = run_felix(felix_system, "Design a REST API", mode="full")

    # Auto mode (Felix decides based on complexity)
    result = run_felix(felix_system, "What time is it?", mode="auto")
"""

import logging
import threading
from typing import Optional, Dict, Any, List, Callable

from src.agents.felix_agent import FelixAgent, FelixResponse

logger = logging.getLogger(__name__)


def run_felix(
    felix_system,
    user_input: str,
    mode: str = "auto",
    streaming_callback: Optional[Callable] = None,
    knowledge_enabled: bool = True,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    cancel_event: Optional[threading.Event] = None
) -> Dict[str, Any]:
    """
    Unified Felix entry point for all interactions.

    This is THE way to interact with Felix. It replaces:
    - Raw LLM calls (old Simple mode) - use mode="direct"
    - run_felix_workflow() (old Workflow mode) - use mode="full"

    Felix always responds AS Felix, maintaining consistent identity
    regardless of the mode selected.

    Args:
        felix_system: Initialized FelixSystem instance from GUI
        user_input: User's message/task
        mode: Processing mode:
            - "auto": Let Felix classify and route (recommended for CLI)
            - "direct": Force direct inference, no agents (GUI Simple mode)
            - "full": Force full multi-agent orchestration (GUI Workflow mode)
        streaming_callback: Callback for real-time output:
            - For "direct" mode: callback(chunk_text: str)
            - For "full" mode: callback(agent_name: str, chunk_text: str)
        knowledge_enabled: Whether to include knowledge brain context
        conversation_history: Previous messages for context continuity
            Format: [{"role": "user"/"assistant", "content": "..."}]
        cancel_event: Optional threading.Event to signal cancellation

    Returns:
        Dict containing:
            - content: str - Felix's response
            - mode_used: str - "direct", "light", or "full"
            - complexity: str - "SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX"
            - confidence: float - Confidence score (0.0-1.0)
            - thinking_steps: List[Dict] - Agent outputs (for full mode)
            - knowledge_sources: List[str] - Sources used from knowledge brain
            - execution_time: float - Processing time in seconds
            - error: str - Error message if failed

    Example:
        # In GUI chat tab
        result = run_felix(
            felix_system,
            "What files are in the src directory?",
            mode="direct",
            streaming_callback=lambda chunk: display(chunk),
            knowledge_enabled=True
        )
        print(result['content'])
    """
    logger.info(f"run_felix called: mode={mode}, knowledge={knowledge_enabled}")
    logger.debug(f"Input: {user_input[:100]}...")

    try:
        # Create FelixAgent instance
        felix_agent = FelixAgent(felix_system)

        # Process the request
        response: FelixResponse = felix_agent.process(
            message=user_input,
            mode=mode,
            streaming_callback=streaming_callback,
            knowledge_enabled=knowledge_enabled,
            conversation_history=conversation_history,
            cancel_event=cancel_event
        )

        # Convert response to dict
        result = {
            'content': response.content,
            'mode_used': response.mode_used,
            'complexity': response.complexity,
            'confidence': response.confidence,
            'thinking_steps': response.thinking_steps,
            'knowledge_sources': response.knowledge_sources,
            'execution_time': response.execution_time,
            'error': response.error,
            # Backward compatibility with old workflow results
            'centralpost_synthesis': {
                'synthesis_content': response.content,
                'confidence': response.confidence,
                'agents_synthesized': len(response.thinking_steps) if response.thinking_steps else 0,
                'tokens_used': 0,
                'max_tokens': 0,
                'temperature': 0.0,
            },
            'synthesis': response.content,
        }

        logger.info(f"run_felix completed: mode_used={result['mode_used']}, "
                   f"time={result['execution_time']:.2f}s")

        return result

    except Exception as e:
        logger.error(f"run_felix error: {e}", exc_info=True)
        return {
            'content': f"Error: {str(e)}",
            'mode_used': 'error',
            'complexity': 'UNKNOWN',
            'confidence': 0.0,
            'thinking_steps': None,
            'knowledge_sources': None,
            'execution_time': 0.0,
            'error': str(e),
            'centralpost_synthesis': {
                'synthesis_content': f"Error: {str(e)}",
                'confidence': 0.0,
                'agents_synthesized': 0,
                'tokens_used': 0,
                'max_tokens': 0,
                'temperature': 0.0,
            },
            'synthesis': f"Error: {str(e)}",
        }
